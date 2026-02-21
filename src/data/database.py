"""SQLite database with SQLAlchemy ORM for product inventory."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    event,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from src.core.config import CONFIG


# ---------------------------------------------------------------------------
# ORM base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

class Product(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    image_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    images: Mapped[list["ProductImage"]] = relationship(
        "ProductImage", back_populates="product", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Product id={self.id} name={self.name!r}>"


class ProductImage(Base):
    __tablename__ = "product_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("products.id", ondelete="CASCADE"), nullable=False
    )
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    source_video: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    frame_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    product: Mapped["Product"] = relationship("Product", back_populates="images")

    def __repr__(self) -> str:
        return f"<ProductImage id={self.id} product_id={self.product_id}>"


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_type: Mapped[str] = mapped_column(String, nullable=False)  # 'incremental' or 'full'
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    epochs: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    best_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    def __repr__(self) -> str:
        return f"<TrainingRun id={self.id} type={self.run_type!r} status={self.status!r}>"


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_type: Mapped[str] = mapped_column(String, nullable=False)  # 'knn' or 'classifier'
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    num_classes: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    def __repr__(self) -> str:
        return (
            f"<ModelRegistry id={self.id} type={self.model_type!r} "
            f"active={self.is_active}>"
        )


# ---------------------------------------------------------------------------
# DatabaseManager
# ---------------------------------------------------------------------------

class DatabaseManager:
    """Thin wrapper around SQLAlchemy engine + session creation."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = str(CONFIG.db_path)
        self.db_path = db_path
        self._engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            future=True,
        )
        # Enable SQLite foreign-key enforcement
        @event.listens_for(self._engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)
        self._session: Session | None = None

    # -- table management ---------------------------------------------------

    def create_tables(self) -> None:
        """Create all tables if they do not already exist."""
        Base.metadata.create_all(self._engine)

    # -- session helpers ----------------------------------------------------

    def get_session(self) -> Session:
        """Return a new session (caller is responsible for closing it)."""
        return self._session_factory()

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> Session:
        self._session = self.get_session()
        return self._session

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._session is None:
            return
        try:
            if exc_type is None:
                self._session.commit()
            else:
                self._session.rollback()
        finally:
            self._session.close()
            self._session = None


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def create_product(session: Session, name: str, description: str | None = None) -> Product:
    """Insert a new product and return it."""
    product = Product(name=name, description=description)
    session.add(product)
    session.flush()
    return product


def get_product(session: Session, product_id: int) -> Product | None:
    """Return a single product by id, or ``None``."""
    return session.get(Product, product_id)


def get_all_products(session: Session) -> list[Product]:
    """Return every product ordered by name."""
    return list(session.query(Product).order_by(Product.name).all())


def update_product(session: Session, product_id: int, **kwargs) -> Product:
    """Update product fields and return the updated product.

    Raises ``ValueError`` if the product does not exist.
    """
    product = session.get(Product, product_id)
    if product is None:
        raise ValueError(f"Product {product_id} not found")
    for key, value in kwargs.items():
        if not hasattr(product, key):
            raise AttributeError(f"Product has no attribute {key!r}")
        setattr(product, key, value)
    product.updated_at = _utcnow()
    session.flush()
    return product


def delete_product(session: Session, product_id: int) -> bool:
    """Delete a product by id.  Returns ``True`` if it existed."""
    product = session.get(Product, product_id)
    if product is None:
        return False
    session.delete(product)
    session.flush()
    return True


def add_product_image(
    session: Session,
    product_id: int,
    file_path: str,
    source_video: str | None = None,
    frame_index: int | None = None,
) -> ProductImage:
    """Add an image record for a product and bump ``image_count``."""
    image = ProductImage(
        product_id=product_id,
        file_path=file_path,
        source_video=source_video,
        frame_index=frame_index,
    )
    session.add(image)
    product = session.get(Product, product_id)
    if product is not None:
        product.image_count = (product.image_count or 0) + 1
        product.updated_at = _utcnow()
    session.flush()
    return image


def get_product_images(session: Session, product_id: int) -> list[ProductImage]:
    """Return all images for a given product."""
    return list(
        session.query(ProductImage)
        .filter(ProductImage.product_id == product_id)
        .order_by(ProductImage.id)
        .all()
    )


def create_training_run(session: Session, run_type: str) -> TrainingRun:
    """Insert a new training run with status ``pending``."""
    run = TrainingRun(run_type=run_type, status="pending")
    session.add(run)
    session.flush()
    return run


def update_training_run(session: Session, run_id: int, **kwargs) -> TrainingRun:
    """Update a training run and return it.

    Raises ``ValueError`` if the run does not exist.
    """
    run = session.get(TrainingRun, run_id)
    if run is None:
        raise ValueError(f"TrainingRun {run_id} not found")
    for key, value in kwargs.items():
        if not hasattr(run, key):
            raise AttributeError(f"TrainingRun has no attribute {key!r}")
        setattr(run, key, value)
    session.flush()
    return run


def get_active_model(session: Session, model_type: str) -> ModelRegistry | None:
    """Return the currently active model for the given type, or ``None``."""
    return (
        session.query(ModelRegistry)
        .filter(ModelRegistry.model_type == model_type, ModelRegistry.is_active.is_(True))
        .first()
    )


def register_model(
    session: Session,
    model_type: str,
    file_path: str,
    num_classes: int,
    activate: bool = True,
) -> ModelRegistry:
    """Register a new model.  If *activate* is True, deactivate all other
    models of the same type first."""
    if activate:
        session.query(ModelRegistry).filter(
            ModelRegistry.model_type == model_type,
            ModelRegistry.is_active.is_(True),
        ).update({"is_active": False})
    model = ModelRegistry(
        model_type=model_type,
        file_path=file_path,
        num_classes=num_classes,
        is_active=activate,
    )
    session.add(model)
    session.flush()
    return model
